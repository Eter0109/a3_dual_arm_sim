"""Measured object frames and a goal-bound version of the native contact skill.

This is an explicitly privileged reference: pose/contact measurements come from
the simulator. It does not turn an unconditioned neural checkpoint into a skill.
"""
import numpy as np
from .motion import average_rotation, group_width


def group_frame(scene, ids):
    positions = np.asarray([scene.cookies[i].position for i in ids], dtype=float)
    rotation = average_rotation([scene.cookies[i].rotation for i in ids])
    if not np.isfinite(positions).all():
        raise ValueError("Non-finite source geometry")
    center = positions.mean(0)
    return center, rotation, (positions - center) @ rotation


def validate_pick_group(scene, ids, max_face_gap=.012):
    if len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError("A pick requires exactly five distinct objects")
    if any(not scene.cookies[i].in_source for i in ids):
        raise ValueError("The selected source group is no longer available")
    if set(ids) & scene.held_ids:
        raise ValueError("The selected source group is already in contact with both jaws")
    center, rotation, local = group_frame(scene, ids)
    order = np.argsort(local[:, 1])
    radii = np.asarray([abs(rotation[:, 1] @ scene.cookies[i].rotation) @ scene.half_size
                        for i in ids])
    gaps = np.diff(local[order, 1]) - radii[order[:-1]] - radii[order[1:]]
    if np.any(gaps > max_face_gap):
        raise ValueError("The source group contains a physical gap")
    if np.ptp(local[:, 0]) > .018 or np.ptp(local[:, 2]) > .018:
        raise ValueError("The selected five are too scattered for one parallel-jaw pick")
    if group_width(scene, ids, rotation[:, 1]) > .080:
        raise ValueError("The selected group exceeds the gripper's usable width")
    return center, rotation, local


class GoalBoundSelection:
    """Mixin: the contact controller may execute only the measured requested group."""
    @property
    def _is_followup_batch(self):
        return bool(self.completed_cookie_indices)

    def _candidate_batches(self):
        goal = getattr(self, "skill_goal", None)
        if goal is None:
            raise RuntimeError("The contact skill has no bound batch goal")
        if any(i in self.completed_cookie_indices for i in goal.cookie_ids):
            raise RuntimeError("A completed object was requested for picking")
        if all(self.env.privileged_cookie_in_source(i) for i in goal.cookie_ids):
            yield list(goal.cookie_ids)


class GroundedBatchExecutor:
    """One measured pick primitive, reused with a fresh goal on every batch.

    Native insertion/force/slip safeguards are retained. Source selection is
    overridden before any pick command, never after an unintended group lifts.
    """
    name = "goal_bound_contact_skill"
    goal_grounded = True
    goal_binding = "measured_object_frame"

    def __init__(self, env, servo, config, seed):
        self.env, self.servo, self.config, self.seed = env, servo, config, seed
        self.scene = None
        self.expert = None
        self.starts = 0

    def update_scene(self, scene):
        self.scene = scene

    def start(self, goal, completed, observation):
        from a3_dual_arm_sim.contracts import EpisodeContext
        from a3_dual_arm_sim.controllers.same_column_batch_expert import A3VariedColumnBatchExpert
        from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
        center, rotation, _ = validate_pick_group(self.scene, goal.cookie_ids)
        source = np.asarray(self.env.SOURCE_POSITIONS)
        columns = np.unique(source[:, 0])
        ranks = {int(np.argmin(abs(columns - source[i, 0]))) for i in goal.cookie_ids}
        if len(ranks) != 1:
            raise ValueError("A bound pick cannot span source columns")
        column = ranks.pop()
        base = A3Cookie2x10Expert if self.env.task_config.required_cookies == 20 else A3VariedColumnBatchExpert
        bound = type("GoalBound" + base.__name__, (GoalBoundSelection, base), {})
        kwargs = dict(first_grasp=5, sampling_seed=self.seed,
                      requested_source_column_index=column) if base is A3Cookie2x10Expert else {}
        self.expert = bound(self.env, **kwargs)
        self.expert.requested_source_column_index = column
        self.expert.reset(EpisodeContext(self.seed, goal.instruction, "cartesian_delta"))
        self.expert.skill_goal = goal
        self.expert.batch_index = goal.index
        self.expert.completed_cookie_indices = list(completed)
        self.goal, self.completed = goal, tuple(completed)
        self.initial_center, self.initial_rotation = center.copy(), rotation.copy()
        self.relocalized = 0
        self.starts += 1
        self.diagnostics = dict(goal_binding="measured_object_frame", selected_ids=list(goal.cookie_ids),
            source_center_world=center.tolist(), source_rotation_world=rotation.tolist(),
            entry_kind="initial" if not completed else "after_previous_release", stage="SELECT_COOKIE",
            physical_resets=0, relocalizations=0)

    def ready(self, scene):
        if self.expert.phase.name != "MOVE_TO_SLOT":
            return False
        if not set(self.goal.cookie_ids) <= scene.held_ids:
            raise RuntimeError("The requested five did not remain in the two-jaw contact chain")
        outside = {i for i in scene.held_ids if not scene.cookies[i].in_source}
        if outside - set(self.goal.cookie_ids):
            raise RuntimeError("An unintended extra object lifted with the bound group")
        return all(not scene.cookies[i].in_source for i in self.goal.cookie_ids)

    def act(self, observation):
        from a3_dual_arm_sim.controllers.expert import CookiePhase
        scene, expert = self.scene, self.expert
        # Refresh safe approach geometry if the measured group moved. Do not
        # chase pose noise after insertion/contact has started.
        if expert.phase.name in ("APPROACH", "PRE_CLOSE"):
            center, rotation, _ = validate_pick_group(scene, self.goal.cookie_ids)
            shift = float(np.linalg.norm(center - self.initial_center))
            turn = float(np.linalg.norm(rotation - self.initial_rotation))
            if shift > .003 or turn > .10:
                if self.relocalized >= 3:
                    raise RuntimeError("Source geometry kept changing during approach")
                expert._advance(CookiePhase.SELECT_COOKIE)
                self.initial_center, self.initial_rotation = center.copy(), rotation.copy()
                self.relocalized += 1
        action = expert.act(observation, self.goal.instruction)
        if expert.failed:
            raise RuntimeError(expert.failure_reason)
        if expert.batch_indices and tuple(expert.batch_indices) != self.goal.cookie_ids:
            raise RuntimeError("The contact primitive violated its bound batch goal")
        self.diagnostics.update(stage=expert.phase.name, relocalizations=self.relocalized,
                                max_pad_force_n=float(expert.max_pad_force))
        return self.servo.canonical(action)

    def pause(self):
        pass
