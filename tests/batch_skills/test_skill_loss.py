import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch not installed in this lightweight environment")
class LossMaskTests(unittest.TestCase):
    def test_visual_batch_reset_does_not_restart_torch_noise(self):
        import torch
        from a3_batch_skills.visual_goals import VisualGoalPolicy,VISUAL_MARKER_VERSION
        from a3_batch_skills.rgbd_localization import LOCALIZER_VERSION
        class Legacy:
            def reset(self,context):torch.manual_seed(123)
        wrapper=VisualGoalPolicy(Legacy(),dict(goal_conditioning=VISUAL_MARKER_VERSION,
            localization=LOCALIZER_VERSION,training_completed=True))
        torch.manual_seed(89);before=torch.get_rng_state().clone()
        wrapper.start(object());self.assertTrue(torch.equal(before,torch.get_rng_state()))
        wrapper.pause();self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_padding_and_inactive_dimensions_have_no_training_gradient(self):
        import torch
        from a3_batch_skills.training import masked_flow_loss
        values = torch.ones(2,3,32,requires_grad=True)
        mask = torch.tensor([[False,True,True],[False,False,True]])
        loss = masked_flow_loss(values,mask).mean()
        self.assertAlmostEqual(loss.item(),1.)
        loss.backward()
        self.assertEqual(values.grad[:,:,8:].abs().sum().item(),0.)
        self.assertEqual(values.grad[0,1:].abs().sum().item(),0.)
        self.assertEqual(values.grad[1,2:].abs().sum().item(),0.)
        self.assertGreater(values.grad[0,0,:8].sum().item(),0.)

    def test_extreme_padding_values_do_not_change_real_action_loss(self):
        import torch
        from a3_batch_skills.training import masked_flow_loss
        values = torch.ones(1,2,32)
        values[:,1,:] = 100000.
        values[:,:,8:] = 200000.
        self.assertEqual(masked_flow_loss(values,torch.tensor([[False,True]])).item(),1.)
        with self.assertRaises(ValueError):
            masked_flow_loss(values,torch.ones(1,2,dtype=torch.bool))

    def test_installed_pi05_forward_uses_the_time_and_dimension_mask(self):
        import torch
        from types import SimpleNamespace as NS
        import lerobot.policies.pi05.modeling_pi05 as module
        from a3_batch_skills.training import install_pi05_skill_loss
        original = module.PI05Policy.forward
        try:
            install_pi05_skill_loss()
            values = torch.ones(1,3,32,requires_grad=True)
            batch = {module.OBS_LANGUAGE_TOKENS:torch.zeros(1,1,dtype=torch.long),
                     module.OBS_LANGUAGE_ATTENTION_MASK:torch.ones(1,1,dtype=torch.bool),
                     "action_is_pad":torch.tensor([[False,True,True]])}
            fake = NS(config=NS(output_features={module.ACTION:NS(shape=[8])}),
                      _preprocess_images=lambda batch:([],[]),prepare_action=lambda batch:None,
                      model=NS(forward=lambda *args:values))
            loss, metrics = module.PI05Policy.forward(fake,batch)
            self.assertEqual(loss.item(),1.)
            self.assertEqual(len(metrics["loss_per_dim"]),8)
            loss.backward()
            self.assertEqual(values.grad[:,1:].abs().sum().item(),0.)
            self.assertEqual(values.grad[:,:,8:].abs().sum().item(),0.)
        finally:
            module.PI05Policy.forward = original


if __name__ == "__main__":
    unittest.main()
