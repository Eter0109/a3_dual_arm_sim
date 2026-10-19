"""Observe -> transfer -> measured placement -> verify/correct -> next batch."""
from dataclasses import asdict
import numpy as np
from .contracts import BatchConfig, BatchGoal, Phase
from .planning import BatchPlanner
from .placement import PrecisePlacement, AdjustPlacement, AlignPickEntry
from .verification import inspect_placement


class BatchSkillController:
    action_mode = "joint_position"

    def __init__(self, env, provider, servo, executor, *, config=None,
                 exercise_adjustment=False):
        self.env, self.provider, self.servo, self.executor = env, provider, servo, executor
        self.config = config or BatchConfig()
        self.planner = BatchPlanner(env, source_column=self.config.source_column)
        self.phase = Phase.LOCATE
        self.batch_index = 0
        self.completed = []
        self.goals = []
        self.events = []
        self.step = 0
        self.failure_reason = None
        self.exercise_adjustment = exercise_adjustment
        self._exercised = False
        self.report = None

    @property
    def finished(self):
        return self.phase in (Phase.DONE, Phase.FAILED)

    def _event(self, name, **kwargs):
        self.events.append(dict(event=name, step=self.step, batch=self.batch_index + 1, **kwargs))

    def _fail(self, reason):
        self.failure_reason, self.phase = str(reason), Phase.FAILED
        self.executor.pause()
        self._event("failed", reason=self.failure_reason)

    def _verify_start(self):
        self.phase, self.verify_age, self.stable = Phase.VERIFY, 0, 0

    def _begin_adjustment(self, goal, scene, reason):
        if not self.config.adjust_placement or self.adjustments >= self.config.max_adjustments:
            raise RuntimeError("Placement correction disabled or retry budget exhausted: " + reason)
        self.executor.pause()
        self.adjustments += 1
        self.adjuster = AdjustPlacement(self.servo, self.config)
        self.adjuster.start(goal, scene)
        if self.adjuster.failure:
            raise RuntimeError(self.adjuster.failure)
        self.phase = Phase.ADJUST
        before_error = float(np.linalg.norm(scene.local(goal.cookie_ids).mean(0)[:2]
                                            - goal.target_positions.mean(0)[:2]))
        self._event("adjustment_started", reason=reason, object_ids=list(goal.cookie_ids),
                    target_positions=goal.target_positions.tolist(), before_error_m=before_error)

    def act(self, observation, native_info=None):
        if self.finished:
            return self.servo.hold()
        self.step += 1
        try:
            scene = self.provider.observe(observation)
            if scene.safety_stop:
                raise RuntimeError("Environment safety stop")
            if self.phase != Phase.LOCATE and self.step - self.batch_started > self.config.max_batch_steps:
                raise RuntimeError("Batch exceeded its total control-step budget")
            if self.phase == Phase.LOCATE:
                self.goal = self.planner.next_goal(self.batch_index, self.completed, scene)
                self.batch_started, self.adjustments = self.step, 0
                # The only reset here is executor state; never env.reset().
                if self.config.align_policy_entry and getattr(self.executor, "requires_prealign", False):
                    self.pick_aligner = AlignPickEntry(self.servo, self.config)
                    self.pick_aligner.start(self.goal, scene)
                    self.phase = Phase.ALIGN_PICK
                else:
                    self.executor.start(self.goal, self.completed, observation)
                    self.phase = Phase.TRANSFER
                self._event("batch_started", object_ids=list(self.goal.cookie_ids),
                    preserved_completed=list(self.completed), localization=scene.source,
                    target_positions=self.goal.target_positions.tolist())
                return self.servo.hold()
            if self.phase == Phase.ALIGN_PICK:
                command = self.pick_aligner.act(scene)
                if self.pick_aligner.failure:
                    raise RuntimeError(self.pick_aligner.failure)
                if self.pick_aligner.done:
                    self.executor.start(self.goal, self.completed, observation)
                    self.phase = Phase.TRANSFER
                    self._event("pick_entry_aligned", object_ids=list(self.goal.cookie_ids))
                return command
            if self.phase == Phase.TRANSFER:
                observe_batch = getattr(self.executor, "observed_batch", None)
                actual = observe_batch(scene, self.completed) if observe_batch else None
                if actual is not None and set(actual) != set(self.goal.cookie_ids):
                    if not set(actual) <= self.planner.allowed_ids:
                        raise RuntimeError("Policy moved objects outside the requested source column; unrestricted tasks may use --source-column auto")
                    self.goal = BatchGoal(self.goal.index, actual, self.goal.target_positions)
                    self.executor.goal = self.goal
                    self._event("observed_batch_adopted", object_ids=list(actual), localization=scene.source)
                if observe_batch is not None:
                    report = inspect_placement(scene, self.goal, self.completed, self.config)
                    if not report.missing_ids and not (set(self.goal.cookie_ids) & scene.held_ids):
                        # The learned policy can finish placing before a held-
                        # batch handoff is seen. Verify and retreat; do not keep
                        # calling it indefinitely or re-grasp an empty tool.
                        self.executor.pause()
                        self.retreat_position = scene.tool_position + np.array([0., 0., .060])
                        self.retreat_rotation = scene.tool_rotation.copy()
                        self.phase = Phase.RETREAT
                        self._event("policy_released_batch", object_ids=list(self.goal.cookie_ids))
                        return self.servo.hold()
                if self.executor.ready(scene):
                    self.executor.pause()
                    self.placer = PrecisePlacement(self.servo, self.config)
                    self.placer.start(self.goal, scene)
                    self.phase = Phase.ALIGN
                    self._event("placement_handoff", action_queue_cleared=True)
                    return self.servo.hold()
                return self.executor.act(observation)
            if self.phase == Phase.RETREAT:
                command, reached = self.servo.move(self.retreat_position, self.retreat_rotation,
                    float(self.env.last_applied_action[7]), speed=.004)
                if reached:
                    self._verify_start()
                return command
            if self.phase == Phase.ALIGN:
                command = self.placer.act(scene)
                if self.placer.failure:
                    raise RuntimeError(self.placer.failure)
                if self.placer.done:
                    self._event("released_and_retreated")
                    self._verify_start()
                return command
            if self.phase == Phase.ADJUST:
                command = self.adjuster.act(scene)
                if self.adjuster.failure:
                    raise RuntimeError(self.adjuster.failure)
                if self.adjuster.done:
                    goal = self.adjuster.goal
                    error = float(np.linalg.norm(scene.local(goal.cookie_ids).mean(0)[:2]
                                                  - goal.target_positions.mean(0)[:2]))
                    self._event("adjustment_motion_finished", after_error_m=error)
                    self._verify_start()
                return command
            self.verify_age += 1
            next_region = (self.planner.regions[self.batch_index + 1]
                           if self.batch_index + 1 < len(self.planner.regions) else None)
            self.report = inspect_placement(scene, self.goal, self.completed,
                                            self.config, next_region)
            self.stable = self.stable + 1 if self.report.complete else 0
            if self.stable >= self.config.verify_steps:
                if self.exercise_adjustment and not self._exercised:
                    # A developer-only physical correction exercise. Change the
                    # requested pose by 4 mm; do not move simulator objects.
                    self._exercised = True
                    positions = self.goal.target_positions.copy()
                    positions[:, 1] += .004
                    self.goal = BatchGoal(self.goal.index, self.goal.cookie_ids, positions)
                    self._begin_adjustment(self.goal, scene, "explicit_4mm_correction_exercise")
                    return self.servo.hold()
                self.completed.extend(self.goal.cookie_ids)
                self.goals.append(self.goal)
                self._event("batch_completed", completed_ids=list(self.completed),
                    max_goal_error_m=self.report.max_goal_error_m,
                    accepted_ids=sorted(self.report.accepted_ids))
                self.batch_index += 1
                self.phase = Phase.DONE if self.batch_index == len(self.planner.regions) else Phase.LOCATE
                return self.servo.hold()
            if self.verify_age >= self.config.settle_timeout_steps:
                bad = set(self.report.damaged_ids) | set(self.report.blocking_ids)
                target = next((g for g in self.goals if bad & set(g.cookie_ids)), self.goal)
                self._begin_adjustment(target, scene, "placement_or_next_region_check_failed")
            return self.servo.hold()
        except (ValueError, RuntimeError, KeyError) as exc:
            self._fail(str(exc))
            return self.servo.hold()

    def summary(self):
        report = asdict(self.report) if self.report else None
        if report:
            report["accepted_ids"] = sorted(report["accepted_ids"])
        return dict(phase=self.phase.value, completed_ids=list(self.completed),
                    completed_batches=self.batch_index, planned_batches=len(self.planner.regions),
                    steps=self.step, failure_reason=self.failure_reason, events=self.events,
                    placement_report=report, config=asdict(self.config),
                    executor_diagnostics=getattr(self.executor, "diagnostics", None),
                    exercise_adjustment=self.exercise_adjustment)
