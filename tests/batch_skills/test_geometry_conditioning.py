import importlib.util
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
from a3_batch_skills.geometry_context import geometry_features,GEOMETRY_DIM
from a3_batch_skills.rgbd_localization import rotation_z
from a3_batch_skills.rgbd_placement import RobotPlacementState


class GeometryFeaturesTests(unittest.TestCase):
    def setup_scene(self):
        source=np.c_[np.full(5,.1),np.arange(5)*.00633+.3,np.full(5,.77)]
        goal=NS(source_positions=source,source_rotation=np.eye(3),
            target_positions_box=np.zeros((5,3)))
        robot=RobotPlacementState(np.array([.2,.2,.9]),rotation_z(.1),np.zeros(3),.03,.3,2.)
        scene=NS(timestamp=2.,target_box=NS(position=np.array([.1,.1,.77]),rotation=rotation_z(.2)))
        observed=NS(timestamp=2.,center=np.array([.15,.2,.85]),rotation=rotation_z(.3))
        return goal,robot,scene,observed

    def test_geometry_is_invariant_to_rigid_change_of_world_coordinates(self):
        goal,robot,scene,observed=self.setup_scene()
        expected=geometry_features(goal,robot,scene,observed)
        q=rotation_z(.7);shift=np.array([.3,-.2,.4])
        transformed=NS(source_positions=goal.source_positions@q.T+shift,source_rotation=q@goal.source_rotation,
            target_positions_box=goal.target_positions_box)
        r=replace(robot,tool_position=q@robot.tool_position+shift,tool_rotation=q@robot.tool_rotation)
        s=NS(timestamp=2.,target_box=NS(position=q@scene.target_box.position+shift,rotation=q@scene.target_box.rotation))
        o=NS(timestamp=2.,center=q@observed.center+shift,rotation=q@observed.rotation)
        np.testing.assert_allclose(geometry_features(transformed,r,s,o),expected,atol=1e-6)
        self.assertEqual(expected.shape,(GEOMETRY_DIM,))

    def test_missing_or_stale_measurements_are_zero_with_explicit_invalid_flags(self):
        goal,robot,scene,observed=self.setup_scene()
        scene.timestamp=1.;observed.timestamp=1.
        features=geometry_features(goal,robot,scene,observed)
        np.testing.assert_array_equal(features[9:27],0.)
        np.testing.assert_array_equal(features[27:],[1,0,0])
        np.testing.assert_array_equal(geometry_features(None,robot,scene,observed),np.zeros(30))
        scene.timestamp=observed.timestamp=1.95
        np.testing.assert_array_equal(geometry_features(goal,robot,scene,observed)[27:],[1,0,0])

    def test_full_sensor_clip_aligns_observation_and_applied_action_and_checks_native_score(self):
        import tempfile,json
        from pathlib import Path
        from a3_batch_skills.sensor_recording import SensorPolicyClipRecorder
        class Video:
            def __init__(self,*args):pass
            def add(self,*args):pass
            def close(self):pass
        observation={'observation.state':np.arange(16,dtype=float)}
        for name in ('front','left_wrist','right_wrist'):
            observation['observation.images.'+name]=np.zeros((4,4,3),np.uint8)
        for native_count,expected in ((0,False),(5,True)):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'clips';rec=SensorPolicyClipRecorder(root,20,provenance={},video_factory=Video)
                rec.prepare(observation,np.zeros(30),NS(index=0,source_column=1),7,'visual_servo')
                rec.commit(np.arange(8)+.1,visual_verified=True,native_count=native_count)
                rec.close()
                metadata=json.loads((root/'batch_00/metadata.json').read_text())
                self.assertEqual(metadata['clip_success'],expected)
                self.assertEqual(metadata['clip_role'],'hybrid_correction')
                self.assertNotIn('cookie_ids',metadata)
                with np.load(root/'batch_00/trajectory.npz',allow_pickle=False) as rows:
                    np.testing.assert_array_equal(rows['state'][0],np.arange(8))
                    np.testing.assert_allclose(rows['action'][0],np.arange(8)+.1)
                    self.assertEqual(rows['global_step'].tolist(),[7])

    def test_data_gate_requires_corrections_and_later_batch_entries(self):
        from a3_batch_skills.geometry_pipeline import data_gate
        clips=[dict(metadata=dict(frames=400,entry_kind='initial',source_column=1,
            clip_role='teacher_bootstrap')) for _ in range(12)]
        self.assertFalse(data_gate(clips)['passed'])
        for i in range(2):clips[i]['metadata'].update(clip_role='hybrid_correction',visual_servo_frames=100)
        self.assertFalse(data_gate(clips)['passed'])
        for i in range(2,4):clips[i]['metadata'].update(entry_kind='after_previous_release',source_column=2)
        self.assertTrue(data_gate(clips)['passed'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'Requires real PyTorch')
class GeometryHeadTests(unittest.TestCase):
    def setup_head(self):
        import torch
        from a3_batch_skills.geometry_model import attach_geometry_head
        model=torch.nn.Module();model.state_proj=torch.nn.Linear(32,16)
        attach_geometry_head(model)
        return model

    def test_zero_init_preserves_all_old_tokens_and_gets_nonzero_geometry_gradient(self):
        import torch
        from a3_batch_skills.geometry_model import add_geometry_to_state_token
        model=self.setup_head();embedding=torch.randn(2,7,16)
        geometry=torch.zeros(2,30);geometry[:,:9]=.2;geometry[:,27]=1
        result=add_geometry_to_state_token(model,embedding,geometry)
        self.assertTrue(torch.equal(result,embedding))
        result.square().mean().backward()
        self.assertGreater(model.a3_geometry_proj[-1].weight.grad.abs().sum().item(),0.)
        self.assertTrue(torch.isfinite(model.a3_geometry_proj[-1].weight.grad).all())

    def test_original_replay_disables_geometry_even_after_the_head_is_trained(self):
        import torch
        from a3_batch_skills.geometry_model import add_geometry_to_state_token
        model=self.setup_head()
        torch.nn.init.normal_(model.a3_geometry_proj[-1].weight)
        embedding=torch.randn(2,7,16);geometry=torch.zeros(2,30)
        result=add_geometry_to_state_token(model,embedding,geometry)
        self.assertTrue(torch.equal(result,embedding))
        result.sum().backward()
        self.assertEqual(model.a3_geometry_proj[-1].weight.grad.abs().sum().item(),0.)

    def test_goal_changes_update_only_the_state_token_not_camera_or_language_tokens(self):
        import torch
        from a3_batch_skills.geometry_model import add_geometry_to_state_token
        model=self.setup_head();torch.nn.init.normal_(model.a3_geometry_proj[-1].weight)
        embedding=torch.randn(1,7,16);a=torch.zeros(1,30);a[:,28]=1
        b=a.clone();b[:,9]=.2
        out_a=add_geometry_to_state_token(model,embedding,a)
        out_b=add_geometry_to_state_token(model,embedding,b)
        self.assertTrue(torch.equal(out_a[:,:-1],embedding[:,:-1]))
        self.assertGreater((out_a[:,-1]-out_b[:,-1]).abs().sum().item(),0.)
        b[:,28]=0
        with self.assertRaises(ValueError):add_geometry_to_state_token(model,embedding,b)


if __name__=='__main__':unittest.main()
