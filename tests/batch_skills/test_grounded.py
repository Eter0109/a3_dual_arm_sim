import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
import numpy as np
from test_skills import scene_and_goal
from a3_batch_skills.contracts import BatchConfig
from a3_batch_skills.grounding import GoalBoundSelection, group_frame, validate_pick_group
from a3_batch_skills.markers import project_points, draw_box, GoalMarkedPolicy
from a3_batch_skills.recording import SkillClipRecorder
from a3_batch_skills.export import action_chunk_indices
from a3_batch_skills.paired import BoundaryMonitor
from a3_batch_skills.suite import roster, summarize


class GroundingTests(unittest.TestCase):
    def test_exact_goal_reaches_contact_candidate_selection(self):
        scene, goal = scene_and_goal()
        goal = replace(goal,cookie_ids=(7,8,9,10,11))
        bound = GoalBoundSelection()
        bound.skill_goal = goal
        bound.completed_cookie_indices = [2,3,4,5,6]
        bound.env = NS(privileged_cookie_in_source=lambda i:True)
        self.assertEqual(list(bound._candidate_batches()), [[7,8,9,10,11]])
        self.assertTrue(bound._is_followup_batch)
        bound.completed_cookie_indices.append(7)
        with self.assertRaises(RuntimeError):
            list(bound._candidate_batches())

    def test_local_frame_survives_world_translation_and_rotation(self):
        scene, goal = scene_and_goal()
        scene = replace(scene,cookies={i:replace(o,in_source=True) for i,o in scene.cookies.items()})
        c,r,local = validate_pick_group(scene,goal.cookie_ids)
        angle = .42
        rot = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        shift = np.array([.2,-.1,.7])
        moved = replace(scene,cookies={i:replace(o,position=shift+rot@o.position,rotation=rot@o.rotation)
                                       for i,o in scene.cookies.items()})
        cm,rm,lm = validate_pick_group(moved,goal.cookie_ids)
        np.testing.assert_allclose(lm,local,atol=1e-12)
        np.testing.assert_allclose(cm,shift+rot@c,atol=1e-12)
        np.testing.assert_allclose(rm,rot@r,atol=1e-12)

    def test_physical_gap_cannot_be_hidden_by_consecutive_ids(self):
        scene, goal = scene_and_goal()
        objects = {i:replace(o,in_source=True) for i,o in scene.cookies.items()}
        objects[4] = replace(objects[4],position=objects[4].position+[0,.04,0])
        with self.assertRaises(ValueError):
            validate_pick_group(replace(scene,cookies=objects),goal.cookie_ids)

    def test_unavailable_group_is_rejected(self):
        scene, goal = scene_and_goal()
        with self.assertRaises(ValueError):
            validate_pick_group(scene,goal.cookie_ids)


class GoalCueTests(unittest.TestCase):
    def test_projection_uses_negative_camera_z_and_correct_pixel_axes(self):
        uv, visible = project_points([[0,0,-1],[.1,.1,-1],[0,0,1]],np.zeros(3),np.eye(3),90,100,200)
        np.testing.assert_allclose(uv[:2],[[100,50],[105,45]])
        self.assertEqual(visible.tolist(),[True,True,False])

    def test_annotations_do_not_mutate_raw_images_or_draw_behind_camera(self):
        raw = np.zeros((20,20,3),dtype=np.uint8)
        marked = draw_box(raw,np.array([[5,5],[10,10]]),np.array([True,True]),(0,220,255))
        self.assertEqual(raw.sum(),0)
        self.assertGreater(marked.sum(),0)
        invisible = draw_box(raw,np.array([[5,5],[10,10]]),np.array([False,False]),(0,220,255))
        np.testing.assert_array_equal(invisible,raw)

    def test_old_weights_cannot_silently_enable_new_visual_conditioning(self):
        with self.assertRaises(ValueError):
            GoalMarkedPolicy(NS(),None,None,{})


class RecordingTests(unittest.TestCase):
    def test_pre_action_state_applied_action_and_batch_boundaries(self):
        class FakeVideo:
            def __init__(self,*args):
                self.frames = []
            def add(self,image):
                self.frames.append(image.copy())
            def close(self):
                pass
        scene, goal = scene_and_goal()
        obs = {"observation.state":np.arange(16,dtype=float)}
        obs.update({"observation.images."+name:np.zeros((10,10,3),dtype=np.uint8)
                    for name in ("front","left_wrist","right_wrist")})
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            recorder = SkillClipRecorder(Path(tmp)/"clips",NS(config=NS(control_hz=20)),
                                        seed=3,marked=False,video_factory=FakeVideo)
            recorder.prepare(obs,goal,scene,100,"LOCATE")
            obs["observation.state"][:] = 99
            recorder.commit(np.ones(8)*.2,batch_completed=True)
            second = replace(goal,index=1)
            recorder.prepare(obs,second,scene,101,"LOCATE")
            recorder.commit(np.ones(8)*.3,failed=True,reason="test rejection")
            recorder.close()
            with np.load(Path(tmp)/"clips/batch_00/trajectory.npz") as trace:
                np.testing.assert_array_equal(trace["state"][0],np.arange(8))
                np.testing.assert_allclose(trace["action"][0],.2)
                self.assertEqual(trace["global_step"].tolist(),[100])
            meta = json.loads((Path(tmp)/"clips/manifest.json").read_text())
            self.assertEqual([m["clip_success"] for m in meta["clips"]],[True,False])
            self.assertEqual(meta["clips"][1]["entry_kind"],"after_previous_release")
            self.assertEqual(meta["clips"][0]["action_dim"],8)

    def test_future_action_chunks_never_cross_the_batch_boundary(self):
        indices, mask = action_chunk_indices(length=7,start=5,horizon=5)
        self.assertEqual(indices.tolist(),[5,6,6,6,6])
        self.assertEqual(mask.tolist(),[False,False,True,True,True])
        with self.assertRaises(ValueError):
            action_chunk_indices(0,0,16)


class ProtocolTests(unittest.TestCase):
    def test_boundary_queue_clear_requires_release_stability_and_retreat(self):
        scene,_ = scene_and_goal()
        monitor = BoundaryMonitor(10,stable_steps=2)
        first,done,_ = monitor.observe(scene,BatchConfig(),[0,0,.02],0)
        second,done,_ = monitor.observe(scene,BatchConfig(),[0,0,.02],1)
        self.assertIsNone(first)
        self.assertIsNone(second)
        third,done,_ = monitor.observe(scene,BatchConfig(),[0,0,.13],2)
        self.assertEqual(third["cookie_ids"],[0,1,2,3,4])
        self.assertFalse(done)
        fourth,_,_ = monitor.observe(scene,BatchConfig(),[0,0,.13],3)
        self.assertIsNone(fourth)

    def test_two_and_four_batches_receive_equal_per_batch_budgets(self):
        cases = roster(["2x5","2x10"],["grounded"],[1,2],["1"],"basic")
        self.assertEqual([r["max_steps"] for r in cases],[1500,1500,3000,3000])
        self.assertEqual({r["batch_steps"] for r in cases},{750})
        rows = [dict(**c,selected_success=True,process_error=False) for c in cases]
        self.assertEqual([g["successes"] for g in summarize(rows).values()],[2,2])


if __name__ == "__main__":
    unittest.main()
