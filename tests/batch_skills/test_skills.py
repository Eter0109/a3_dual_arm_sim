import unittest
from unittest.mock import patch
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
from a3_batch_skills.contracts import BatchConfig, BatchGoal, CookieState, SceneState, Phase
from a3_batch_skills.planning import target_regions, BatchPlanner
from a3_batch_skills.verification import inspect_placement, accepted_ids
from a3_batch_skills.motion import LeftArmServo
from a3_batch_skills.placement import AdjustPlacement
from a3_batch_skills.perception import ExternalSceneProvider
from a3_batch_skills.controller import BatchSkillController


HALF = np.array([.025, .0095/3, .0125])


def slots(rows=5):
    y = (np.arange(rows) - (rows - 1)/2) * 2*HALF[1]
    return np.asarray([(x, v) for x in (-.028, .028) for v in y])


def scene_and_goal():
    targets = target_regions(slots(), .0185)[0]
    goal = BatchGoal(0, tuple(range(5)), targets)
    objects = {i: CookieState(i, p.copy(), np.eye(3), 0., 0., False, False)
               for i, p in enumerate(targets)}
    scene = SceneState(objects, np.zeros(3), np.eye(3), np.array([0., 0., .13]), np.eye(3),
        HALF, np.array([.057, .029]), .006, .026, frozenset(), (0., 0.), False, "test_estimate")
    return scene, goal


class PlanningTests(unittest.TestCase):
    def test_two_and_four_batches_same_grasp_size(self):
        self.assertEqual([len(g) for g in target_regions(slots(5), .0185)], [5, 5])
        self.assertEqual([len(g) for g in target_regions(slots(10), .0185)], [5]*4)

    def test_larger_box_preserves_release_gap(self):
        regions = target_regions(slots(10), .0185)
        gap = regions[1][0, 1] - regions[0][-1, 1] - 2*HALF[1]
        self.assertAlmostEqual(gap, .008)

    def test_invalid_partial_region_rejected(self):
        with self.assertRaises(ValueError):
            target_regions(slots(7), .0185)

    def test_goal_copies_immutable_positions(self):
        positions = np.ones((5, 3))
        goal = BatchGoal(0, tuple(range(5)), positions)
        positions[:] = 0
        self.assertEqual(float(goal.target_positions.sum()), 15)
        with self.assertRaises(ValueError):
            goal.target_positions[0, 0] = 9

    def test_next_batch_preserves_completed_and_reads_current_source(self):
        env = NS(TARGET_SLOTS_LOCAL=slots(), _target_cookie_center_z=.0185,
            SOURCE_POSITIONS=np.asarray([[0, i*.01] for i in range(20)]),
            task_config=NS(required_cookies=10))
        planner = BatchPlanner(env)
        scene, _ = scene_and_goal()
        objects = {i: CookieState(i, np.zeros(3), np.eye(3), 0, 0, True, False) for i in range(20)}
        scene = replace(scene, cookies=objects)
        completed = [0, 1, 2, 3, 4]
        self.assertEqual(planner.next_goal(1, completed, scene).cookie_ids, (5, 6, 7, 8, 9))
        self.assertEqual(completed, [0, 1, 2, 3, 4])
        objects[6] = replace(objects[6], in_source=False)
        self.assertEqual(planner.next_goal(1, completed, scene).cookie_ids, (7, 8, 9, 10, 11))
        for i in range(5, 20):
            objects[i] = replace(objects[i], in_source=False)
        with self.assertRaises(RuntimeError):
            planner.next_goal(1, completed, scene)

    def test_next_goal_never_combines_cookies_across_a_removed_gap(self):
        env = NS(TARGET_SLOTS_LOCAL=slots(), _target_cookie_center_z=.0185,
            SOURCE_POSITIONS=np.asarray([[0, i*.01] for i in range(20)]),
            task_config=NS(required_cookies=10))
        planner = BatchPlanner(env)
        scene, _ = scene_and_goal()
        objects = {i: CookieState(i, np.zeros(3), np.eye(3), 0, 0, True, False) for i in range(20)}
        goal = planner.next_goal(1, [2, 3, 4, 5, 6], replace(scene, cookies=objects))
        self.assertEqual(goal.cookie_ids, (7, 8, 9, 10, 11))


class VerificationTests(unittest.TestCase):
    def test_contained_and_released_objects_pass(self):
        scene, goal = scene_and_goal()
        report = inspect_placement(scene, goal, (), BatchConfig())
        self.assertTrue(report.complete)
        self.assertEqual(report.accepted_ids, frozenset(range(5)))

    def test_held_and_second_layer_do_not_pass(self):
        scene, _ = scene_and_goal()
        objects = scene.cookies.copy()
        objects[0] = replace(objects[0], finger_contact=True)
        objects[1] = replace(objects[1], position=objects[1].position + [0, 0, .008])
        self.assertEqual(accepted_ids(replace(scene, cookies=objects), BatchConfig()), frozenset({2, 3, 4}))

    def test_damage_to_previous_objects_is_checked(self):
        scene, goal = scene_and_goal()
        objects = scene.cookies.copy()
        objects[7] = replace(objects[0], identifier=7, position=np.array([1., 0, .0185]))
        report = inspect_placement(replace(scene, cookies=objects), goal, [7], BatchConfig())
        self.assertEqual(report.damaged_ids, (7,))

    def test_next_region_obstruction_is_detected(self):
        scene, goal = scene_and_goal()
        report = inspect_placement(scene, goal, (), BatchConfig(), goal.target_positions)
        self.assertEqual(report.blocking_ids, tuple(range(5)))
        next_region = target_regions(slots(), .0185)[1]
        self.assertFalse(inspect_placement(scene, goal, (), BatchConfig(), next_region).blocking_ids)

    def test_rigid_world_frame_change_preserves_verdict(self):
        scene, goal = scene_and_goal()
        angle = .7
        r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        t = np.array([.1, .2, .75])
        objects = {i: replace(o, position=t+r@o.position, rotation=r@o.rotation) for i, o in scene.cookies.items()}
        moved = replace(scene, cookies=objects, box_position=t, box_rotation=r)
        self.assertTrue(inspect_placement(moved, goal, (), BatchConfig()).complete)


class AdapterTests(unittest.TestCase):
    def test_left8_actions_hold_actual_initial_right_target(self):
        right = np.array([.2, -.3, .4, .5, -.6, .7, -.8, .9])
        env = NS(last_applied_action=np.r_[np.zeros(8), right], _eef_sites=(0, 1))
        servo = LeftArmServo(env)
        full = servo.canonical(np.arange(16, dtype=float))
        np.testing.assert_array_equal(full[8:], right)
        self.assertEqual(full.shape, (16,))
        with self.assertRaises(ValueError):
            servo.canonical(np.full(8, np.nan))

    def test_adjustment_rejects_scattered_objects_before_moving(self):
        scene, goal = scene_and_goal()
        objects = scene.cookies.copy()
        objects[0] = replace(objects[0], position=objects[0].position + [.2, 0, 0])
        adjuster = AdjustPlacement(NS(), BatchConfig())
        adjuster.start(goal, replace(scene, cookies=objects))
        self.assertIsNotNone(adjuster.failure)

    def test_external_vision_cannot_silently_return_oracle_state(self):
        scene, _ = scene_and_goal()
        provider = ExternalSceneProvider(lambda obs: replace(scene, source="simulator_truth"))
        with self.assertRaises(ValueError):
            provider.observe({})

    def test_balanced_scatter_cannot_hide_behind_unchanged_centroid(self):
        scene, goal = scene_and_goal()
        objects = scene.cookies.copy()
        objects[0] = replace(objects[0], position=objects[0].position + [.1, 0, 0])
        objects[4] = replace(objects[4], position=objects[4].position - [.1, 0, 0])
        adjuster = AdjustPlacement(NS(), BatchConfig())
        adjuster.start(goal, replace(scene, cookies=objects))
        self.assertIsNotNone(adjuster.failure)


class ControllerTests(unittest.TestCase):
    def make_controller(self, config=None):
        scene, _ = scene_and_goal()
        source = np.asarray([[i//20, .2+(i%20)*.01] for i in range(40)])
        objects = {i: CookieState(i, np.r_[source[i], .0185], np.eye(3),
                                 0, 0, True, False) for i in range(40)}
        provider = NS(current=replace(scene, cookies=objects))
        provider.observe = lambda obs: provider.current
        env = NS(TARGET_SLOTS_LOCAL=slots(), _target_cookie_center_z=.0185,
                 SOURCE_POSITIONS=source, task_config=NS(required_cookies=10))
        env.reset = lambda *args, **kw: self.fail("Controller reset live physics")
        executor = NS(starts=[], pause=lambda: None)
        executor.start = lambda goal, completed, obs: executor.starts.append((goal, tuple(completed)))
        servo = NS(hold=lambda *args: np.zeros(16))
        controller = BatchSkillController(env, provider, servo, executor,
                                         config=config or BatchConfig(verify_steps=2))
        return controller, provider, executor

    def place_current_batch(self, controller, provider):
        objects = provider.current.cookies.copy()
        for i, p in zip(controller.goal.cookie_ids, controller.goal.target_positions):
            objects[i] = replace(objects[i], position=p.copy(), in_source=False)
        provider.current = replace(provider.current, cookies=objects)
        controller._verify_start()

    def test_handoff_preserves_progress_and_does_not_reset_physics(self):
        controller, provider, executor = self.make_controller()
        controller.act({})
        self.place_current_batch(controller, provider)
        controller.act({})
        self.assertEqual(controller.completed, [])
        controller.act({})
        self.assertEqual(controller.completed, [0, 1, 2, 3, 4])
        self.assertEqual(controller.phase, Phase.LOCATE)
        controller.act({})
        self.assertEqual(controller.goal.cookie_ids, (5, 6, 7, 8, 9))
        self.assertEqual(executor.starts[-1][1], (0, 1, 2, 3, 4))

    def test_damaged_previous_batch_prevents_completing_next_batch(self):
        config = BatchConfig(verify_steps=1, settle_timeout_steps=1, adjust_placement=False)
        controller, provider, _ = self.make_controller(config)
        controller.act({})
        self.place_current_batch(controller, provider)
        controller.act({})
        controller.act({})
        self.place_current_batch(controller, provider)
        objects = provider.current.cookies.copy()
        objects[0] = replace(objects[0], position=np.array([1., 0, .0185]))
        provider.current = replace(provider.current, cookies=objects)
        controller.act({})
        self.assertEqual(controller.phase, Phase.FAILED)
        self.assertEqual(controller.completed, [0, 1, 2, 3, 4])

    def test_explicit_auto_mode_adopts_actual_released_batch(self):
        controller, provider, executor = self.make_controller(BatchConfig(source_column="auto"))
        controller.act({})
        moved = (20, 21, 22, 23, 24)
        objects = provider.current.cookies.copy()
        for i, p in zip(moved, controller.goal.target_positions):
            objects[i] = replace(objects[i], position=p.copy(), in_source=False)
        provider.current = replace(provider.current, cookies=objects)
        executor.observed_batch = lambda scene, completed: moved
        controller.act({})
        self.assertEqual(controller.goal.cookie_ids, moved)
        self.assertEqual(controller.phase, Phase.RETREAT)
        self.assertTrue(any(e["event"] == "policy_released_batch" for e in controller.events))

    def test_fixed_source_never_silently_adopts_other_objects(self):
        controller, provider, executor = self.make_controller()
        controller.act({})
        executor.observed_batch = lambda scene, completed: (20, 21, 22, 23, 24)
        controller.act({})
        self.assertEqual(controller.phase, Phase.FAILED)
        self.assertEqual(controller.completed, [])

    def test_fixed_column_can_track_actual_five_within_that_column(self):
        controller, provider, executor = self.make_controller()
        controller.act({})
        moved = (2, 3, 4, 5, 6)
        objects = provider.current.cookies.copy()
        for i, p in zip(moved, controller.goal.target_positions):
            objects[i] = replace(objects[i], position=p.copy(), in_source=False)
        provider.current = replace(provider.current, cookies=objects)
        executor.observed_batch = lambda scene, completed: moved
        controller.act({})
        self.assertEqual(controller.goal.cookie_ids, moved)
        self.assertEqual(controller.phase, Phase.RETREAT)

    def test_policy_restart_happens_after_pregrasp_alignment(self):
        controller, provider, executor = self.make_controller(BatchConfig(align_policy_entry=True))
        executor.requires_prealign = True

        class FakeAligner:
            def __init__(self, servo, config):
                self.done, self.failure = False, None

            def start(self, goal, scene):
                self.goal = goal

            def act(self, scene):
                self.done = True
                return np.zeros(16)

        with patch("a3_batch_skills.controller.AlignPickEntry", FakeAligner):
            controller.act({})
            self.assertEqual(controller.phase, Phase.ALIGN_PICK)
            self.assertEqual(executor.starts, [])
            controller.act({})
            self.assertEqual(controller.phase, Phase.TRANSFER)
            self.assertEqual(len(executor.starts), 1)


if __name__ == "__main__":
    unittest.main()
