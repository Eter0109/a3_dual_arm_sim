import numpy as np


class ReferenceBatchExecutor:
    """Current main expert up to the held-batch boundary, with preserved progress."""
    name = "privileged_reference"

    def __init__(self, env, servo, config, seed):
        from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
        from a3_dual_arm_sim.controllers.same_column_batch_expert import (
            A3SameColumnBatchExpert, A3VariedColumnBatchExpert)
        from a3_dual_arm_sim.contracts import EpisodeContext
        self.env, self.servo, self.config = env, servo, config
        column = 1 if config.source_column == "auto" else config.source_column
        if env.task_config.required_cookies == 20:
            self.expert = A3Cookie2x10Expert(env, first_grasp=5, sampling_seed=seed,
                requested_source_column_index=column - 1)
        else:
            cls = A3SameColumnBatchExpert if column == 1 else A3VariedColumnBatchExpert
            self.expert = cls(env)
            self.expert.requested_source_column_index = column - 1
        self.expert.reset(EpisodeContext(seed, "Repeat five-cookie transfers.", "cartesian_delta"))

    def start(self, goal, completed, observation):
        from a3_dual_arm_sim.controllers.expert import CookiePhase
        expert = self.expert
        expert.batch_index = goal.index
        expert.completed_cookie_indices = list(completed)
        expert.batch_indices = []
        expert.failure_reason = expert.failure_phase = None
        expert._base_push_attempts = 0
        expert._held_rotation_batch = None
        expert._grip_reference = None
        expert._advance(CookiePhase.SELECT_COOKIE)
        self.goal = goal

    def ready(self, scene):
        return self.expert.phase.name == "MOVE_TO_SLOT"

    def act(self, observation):
        action = self.expert.act(observation, self.goal.instruction)
        if self.expert.failed:
            raise RuntimeError(self.expert.failure_reason)
        if self.expert.batch_indices and tuple(self.expert.batch_indices) != self.goal.cookie_ids:
            raise RuntimeError("Reference executor selected a different batch than the scene plan")
        return self.servo.canonical(action)

    def pause(self):
        pass  # no neural action queue


class PolicyBatchExecutor:
    """One policy instance reused per batch; no learned right-arm outputs are applied."""
    name = "learned_left8_with_geometric_placement"
    requires_prealign = True

    def __init__(self, policy, servo, config, seed, instruction_override=None):
        self.policy, self.servo, self.config, self.seed = policy, servo, config, seed
        self.instruction_override = instruction_override
        self.goal_grounded = callable(getattr(policy, "set_batch_goal", None))
        self.goal_binding = "projected_target_cues" if self.goal_grounded else "unconditioned_policy"

    @property
    def instruction(self):
        return self.instruction_override or self.goal.instruction

    def start(self, goal, completed, observation):
        from a3_dual_arm_sim.contracts import EpisodeContext
        self.goal = goal
        self.policy.reset(EpisodeContext(self.seed, self.instruction, self.policy.action_mode))
        hook = getattr(self.policy, "set_batch_goal", None)
        if hook is not None:
            hook(goal)

    def ready(self, scene):
        if not set(self.goal.cookie_ids) <= scene.held_ids:
            return False
        if self.config.policy_scope == "pick":
            return all(not scene.cookies[i].in_source for i in self.goal.cookie_ids)
        mean = np.asarray([scene.cookies[i].position for i in self.goal.cookie_ids]).mean(0)
        target = scene.world(self.goal.target_positions.mean(0))
        return np.linalg.norm(mean[:2] - target[:2]) <= self.config.takeover_distance_m

    def observed_batch(self, scene, completed):
        from .verification import accepted_ids
        held = sorted(i for i in scene.held_ids if i not in completed and not scene.cookies[i].in_source)
        released = sorted(accepted_ids(scene, self.config) - set(completed))
        self.diagnostics = dict(expected_ids=list(self.goal.cookie_ids), held_outside_ids=held,
                                released_candidate_ids=released,
                                finger_contact_ids=[i for i, o in scene.cookies.items() if o.finger_contact])
        if len(held) == 5:
            return tuple(held)
        if len(released) == 5:
            return tuple(released)
        return None

    def act(self, observation):
        action = self.policy.act(observation, self.instruction)
        return self.servo.canonical(action)

    def pause(self):
        from a3_dual_arm_sim.contracts import EpisodeContext
        goal = getattr(self, "goal", None)
        self.policy.reset(EpisodeContext(self.seed, self.instruction if goal else "",
                                         self.policy.action_mode))
