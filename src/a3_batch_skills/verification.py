from dataclasses import dataclass
import numpy as np
from .planning import rectangle_overlap


@dataclass(frozen=True)
class PlacementReport:
    accepted_ids: frozenset[int]
    missing_ids: tuple[int, ...]
    damaged_ids: tuple[int, ...]
    blocking_ids: tuple[int, ...]
    max_goal_error_m: float

    @property
    def complete(self):
        return not (self.missing_ids or self.damaged_ids or self.blocking_ids)


def accepted_ids(scene, config):
    if config.floor_clearance_m >= 2 * min(scene.half_size):
        raise ValueError("Floor clearance must be below a full cookie thickness")
    accepted = set()
    for i, obj in scene.cookies.items():
        local = scene.box_rotation.T @ (obj.position - scene.box_position)
        rotation = scene.box_rotation.T @ obj.rotation
        extent = np.abs(rotation) @ scene.half_size
        bottom = local[2] - extent[2]
        finite = np.all(np.isfinite(np.r_[local, extent, obj.linear_speed, obj.angular_speed]))
        inside = (np.all(np.abs(local[:2]) + extent[:2]
                         <= scene.inner_half_size + config.containment_tolerance_m)
                  and scene.floor_top - .003 <= bottom <= scene.floor_top + config.floor_clearance_m)
        upright = abs(rotation[2, 2]) >= np.cos(np.deg2rad(15))
        if (finite and inside and not obj.finger_contact and not scene.safety_stop
                and obj.linear_speed <= .09 and obj.angular_speed <= .30
                and (config.placement_mode != "strict" or upright)):
            accepted.add(i)
    return frozenset(accepted)


def inspect_placement(scene, goal, completed, config, next_region=None):
    accepted = accepted_ids(scene, config)
    blocking = []
    if next_region is not None:
        for i in set(completed) | set(goal.cookie_ids):
            obj = scene.cookies[i]
            local = scene.box_rotation.T @ (obj.position - scene.box_position)
            rotation = scene.box_rotation.T @ obj.rotation
            for target in next_region:
                if rectangle_overlap(local, rotation, scene.half_size,
                                     target, np.eye(3), scene.half_size, margin=-.0005):
                    blocking.append(i)
                    break
    actual = scene.local(goal.cookie_ids)
    # Correspondence is source-order -> target-row order; report geometry of
    # the group independently so tiny internal permutations do not force retries.
    error = float(np.linalg.norm(actual.mean(0)[:2] - goal.target_positions.mean(0)[:2]))
    return PlacementReport(accepted,
        tuple(i for i in goal.cookie_ids if i not in accepted),
        tuple(i for i in completed if i not in accepted), tuple(sorted(blocking)), error)
