"""Left8 flow loss with explicit time padding, installed only in a skill trainer.

Call install_pi05_skill_loss() before the existing state-projector patch. No
inference weights, running trainer, optimizer, or global installation is edited.
"""


def masked_flow_loss(losses, action_is_pad, active_dims=8):
    import torch
    if losses.ndim != 3 or active_dims != 8 or losses.shape[-1] < active_dims:
        raise ValueError("Expected B,T,D flow losses and exactly eight task dimensions")
    if action_is_pad is None or action_is_pad.shape != losses.shape[:2]:
        raise ValueError("Skill training requires a B,T action_is_pad mask")
    valid = ~action_is_pad.to(device=losses.device,dtype=torch.bool)
    count = valid.sum(dim=1)
    if not torch.all(count > 0):
        raise ValueError("A training sample contains no real action labels")
    values = losses[:, :, :active_dims].float()
    values = values.masked_fill(~valid.unsqueeze(-1),0.)
    per_sample = values.sum(dim=(1,2)) / (count*active_dims)
    return per_sample


def install_pi05_skill_loss():
    import torch
    import lerobot.policies.pi05.modeling_pi05 as module
    cls = module.PI05Policy
    if getattr(cls.forward,"_a3_masked_skill_loss",False):
        return

    def forward(self,batch,reduction="mean"):
        if self.config.output_features[module.ACTION].shape[0] != 8:
            raise ValueError("A3 single-batch training requires left8 output features")
        if reduction not in ("mean","none"):
            raise ValueError("Unsupported loss reduction")
        images,image_masks = self._preprocess_images(batch)
        tokens = batch[module.OBS_LANGUAGE_TOKENS]
        masks = batch[module.OBS_LANGUAGE_ATTENTION_MASK]
        losses = self.model.forward(images,image_masks,tokens,masks,self.prepare_action(batch))
        per_sample = masked_flow_loss(losses,batch.get("action_is_pad"))
        loss = per_sample if reduction == "none" else per_sample.mean()
        valid = ~batch["action_is_pad"].to(device=losses.device,dtype=torch.bool)
        per_dim = losses[:,:,:8].float().masked_fill(~valid.unsqueeze(-1),0.).sum(dim=(0,1))/valid.sum()
        return loss,dict(loss=float(per_sample.mean().detach()),
                         loss_per_dim=per_dim.detach().cpu().tolist(),
                         active_action_dimensions=8,action_padding_masked=True)

    forward._a3_masked_skill_loss = True
    cls.forward = forward
